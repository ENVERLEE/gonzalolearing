package com.leareng.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Service;

import jakarta.mail.MessagingException;
import jakarta.mail.internet.MimeMessage;

@Service
public class EmailService {
    
    private final JavaMailSender mailSender;
    
    @Value("${spring.mail.username}")
    private String fromEmail;
    
    public EmailService(JavaMailSender mailSender) {
        this.mailSender = mailSender;
    }
    
    public void sendEmail(String to, String subject, String body) {
        try {
            MimeMessage message = mailSender.createMimeMessage();
            MimeMessageHelper helper = new MimeMessageHelper(message, true, "UTF-8");
            
            helper.setFrom(fromEmail);
            helper.setTo(to);
            helper.setSubject(subject);
            helper.setText(body, true); // true indicates HTML
            
            mailSender.send(message);
        } catch (MessagingException e) {
            throw new RuntimeException("Failed to send email", e);
        }
    }
    
    public void sendVerificationEmail(String to, String token) {
        String subject = "Leareng 이메일 인증";
        String body = String.format("""
            <h2>Leareng 이메일 인증</h2>
            <p>아래 인증 코드를 입력하여 이메일을 인증해주세요:</p>
            <h3 style="background-color: #f0f0f0; padding: 10px; font-family: monospace;">%s</h3>
            <p>이 인증 코드는 24시간 동안 유효합니다.</p>
            """, token);
        sendEmail(to, subject, body);
    }
    
    public void sendPasswordResetEmail(String to, String token) {
        String subject = "Leareng 비밀번호 재설정";
        String body = String.format("""
            <h2>Leareng 비밀번호 재설정</h2>
            <p>아래 인증 코드를 입력하여 비밀번호를 재설정하세요:</p>
            <h3 style="background-color: #f0f0f0; padding: 10px; font-family: monospace;">%s</h3>
            <p>이 인증 코드는 24시간 동안 유효합니다.</p>
            """, token);
        sendEmail(to, subject, body);
    }
    
    public void sendSubscriptionApprovalEmail(String to) {
        String subject = "Leareng 구독 승인";
        String body = """
            <h2>Leareng 구독 승인</h2>
            <p>회원님의 구독 요청이 승인되었습니다.</p>
            <p>이제 프리미엄 기능을 이용하실 수 있습니다.</p>
            """;
        sendEmail(to, subject, body);
    }
    
    public void sendSubscriptionDenialEmail(String to) {
        String subject = "Leareng 구독 거절 알림";
        String body = """
            <h2>Leareng 구독 거절 알림</h2>
            <p>죄송합니다. 회원님의 구독 요청이 거절되었습니다.</p>
            <p>자세한 사항은 고객센터로 문의해주세요.</p>
            """;
        sendEmail(to, subject, body);
    }
}

